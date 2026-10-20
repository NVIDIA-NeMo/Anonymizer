# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove the corpus-backed witness loaders reproduce the removed witness files exactly."""

from __future__ import annotations

import hashlib

import pytest

from tests.graph_sdk.retirement_witnesses import witness_sources


@pytest.mark.parametrize("name", sorted(witness_sources.ORIGINAL_SHA256))
def test_witness_file_regenerates_byte_identically(name: str) -> None:
    witness_sources.original_bytes.cache_clear()
    first = witness_sources.original_bytes(name)
    witness_sources.original_bytes.cache_clear()
    assert witness_sources.original_bytes(name) == first
    assert hashlib.sha256(first).hexdigest() == witness_sources.ORIGINAL_SHA256[name]


def test_loaders_return_independent_copies() -> None:
    first = witness_sources.workflow_mutant_cases()
    first[0]["declaration"].clear()
    assert witness_sources.workflow_mutant_cases()[0]["declaration"]
