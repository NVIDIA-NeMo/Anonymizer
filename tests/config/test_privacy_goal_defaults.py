# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from anonymizer.config.rewrite import DEFAULT_PRESERVE_TEXT, DEFAULT_PROTECT_TEXT, PrivacyGoal


def test_privacy_goal_fills_the_half_not_given() -> None:
    assert PrivacyGoal(protect="Names, addresses and phone numbers").preserve == DEFAULT_PRESERVE_TEXT
    assert PrivacyGoal(preserve="Clinical meaning and treatment context").protect == DEFAULT_PROTECT_TEXT
    assert PrivacyGoal() == PrivacyGoal(protect=DEFAULT_PROTECT_TEXT, preserve=DEFAULT_PRESERVE_TEXT)
