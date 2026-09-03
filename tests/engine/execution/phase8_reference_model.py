# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure content-free Phase 8 semantic oracle.

It intentionally depends on neither the production Phase 8 modules nor an
adapter/dataframe runtime.  Cases use opaque keys and terminal symbols only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Case:
    groups: tuple[tuple[str, ...], ...]
    atomic_groups: tuple[tuple[str, ...], ...]
    failed_groups: tuple[int, ...] = ()
    embargo: bool = False


def reduce(case: Case) -> tuple[str, ...]:
    """Return ordered released keys under whole-group atomic withholding."""
    if case.embargo:
        return ()
    failed_keys = {key for index in case.failed_groups for key in case.groups[index]}
    withheld = {
        key
        for atomic in case.atomic_groups
        if failed_keys.intersection(atomic)
        for key in atomic
    }
    return tuple(key for group in case.groups for key in group if key not in withheld)
