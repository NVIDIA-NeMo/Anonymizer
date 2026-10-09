# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolation and drift rejection for generated reference fixtures."""

from __future__ import annotations

from typing import Literal

import pytest

from tests.graph_sdk.reference import corpora


@pytest.mark.parametrize("name", ("effects", "qualification"))
def test_corpus_callers_cannot_mutate_another_copy(name: Literal["effects", "qualification"]) -> None:
    first = corpora.load_cases(name)
    second = corpora.load_cases(name)
    first[0]["declaration"].clear()
    first[0]["events"].clear()
    assert corpora.load_cases(name) == second
    assert first != second


@pytest.mark.parametrize("name", ("effects", "qualification"))
def test_changed_generator_cannot_bypass_reviewed_digest(
    name: Literal["effects", "qualification"], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = corpora.effects_v1 if name == "effects" else corpora.qualification_v1
    corpora.corpus_bytes.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(module, "canonical_bytes", lambda cases: b"[]\n")
            with pytest.raises(AssertionError, match="differs from its reviewed manifest"):
                corpora.load_cases(name)
    finally:
        corpora.corpus_bytes.cache_clear()
