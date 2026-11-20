# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A label whose casefold is longer than itself (İ, ẞ) still mints a placeholder is_placeholder and the review flag
recognise (Greptile 4643991385)."""

from __future__ import annotations

import pytest

from anonymizer.engine.replace.llm_replace_workflow import _collision_safe_synthetic, _recase, is_placeholder
from anonymizer.engine.rewrite.rewrite_workflow import _placeholder_in_text


@pytest.mark.parametrize("label", ["İstanbul", "Straẞe", "city"])
def test_minted_placeholder_is_recognised_when_casefold_expands(label: str) -> None:
    minted = _collision_safe_synthetic(label, index=1, protected_original_values=())
    assert is_placeholder(label, minted)
    assert is_placeholder(label, minted.lower())
    assert not is_placeholder(label, minted[:-2] + "x]")


@pytest.mark.parametrize("label", ["İstanbul", "Straẞe"])
def test_review_flag_set_for_a_copied_placeholder_when_casefold_expands(label: str) -> None:
    minted = _collision_safe_synthetic(label, index=1, protected_original_values=())
    replacement_map = {"replacements": [{"original": "Ankara", "label": label, "synthetic": minted}]}
    assert _placeholder_in_text(f"She moved to {minted} last year.", replacement_map)
    assert not _placeholder_in_text("She moved last year.", replacement_map)


@pytest.mark.parametrize("label", ["İstanbul"])
def test_recase_keeps_a_minted_placeholder_when_casefold_expands(label: str) -> None:
    """``_recase`` (label-free ``_is_placeholder``) keeps the token ``is_placeholder`` is pinned on above."""
    minted = _collision_safe_synthetic(label, index=1, protected_original_values=())
    assert _recase(minted, like="ANKARA", source="Ankara") == minted
