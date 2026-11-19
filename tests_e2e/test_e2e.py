# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the Anonymizer rewrite pipeline.

Calls real model endpoints through the bundled default providers
(``src/anonymizer/config/default_model_configs/providers.yaml``), so it needs:

- the API key variable named by the ``openrouter`` provider (``OPENROUTER_API_KEY``), and
- a GLiNER2 detector server at the ``local-gliner2`` endpoint (``http://127.0.0.1:8001/v1``), started
  without a token, for example after ``uv sync --extra notebooks`` (the dev venv alone has no pip):
  ``uv run python tools/serve_gliner.py --host 127.0.0.1 --port 8001``
  (see ``docs/concepts/self-hosting-gliner.md``).

The test skips when either prerequisite is missing.

Usage:
    pytest -m e2e tests_e2e/test_e2e.py
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
import yaml

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite, configure_logging
from anonymizer.engine.ndd.model_loader import DEFAULT_CONFIG_DIR

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


def _default_prerequisites() -> tuple[str, str]:
    """Return the LLM key variable and the detector endpoint named by the bundled providers.yaml."""
    providers = yaml.safe_load((DEFAULT_CONFIG_DIR / "providers.yaml").read_text())["providers"]
    by_name = {provider["name"]: provider for provider in providers}
    return by_name["openrouter"]["api_key"], by_name["local-gliner2"]["endpoint"].rstrip("/")


def _detector_skip_reason(endpoint: str) -> str | None:
    try:
        response = httpx.get(f"{endpoint}/models", timeout=2.0)
    except httpx.HTTPError:
        return f"no GLiNER2 server at {endpoint}; see docs/concepts/self-hosting-gliner.md"
    if response.status_code in (401, 403):
        return f"the GLiNER2 server at {endpoint} requires a token; the bundled providers.yaml sends none"
    return None


@pytest.mark.e2e
def test_rewrite_preview_smoke() -> None:
    key_variable, detector_endpoint = _default_prerequisites()
    if not os.getenv(key_variable):
        pytest.skip(f"{key_variable} is required by the bundled providers.yaml")
    if reason := _detector_skip_reason(detector_endpoint):
        pytest.skip(reason)

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
