# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private-only composition seam for the Phase 8 grouped profile."""

from __future__ import annotations

from collections.abc import Callable

from anonymizer.engine.execution.phase8_runtime import _run_group_operation
from anonymizer.engine.execution.phase8_validation import _Phase8Metric


class _Phase8GroupedRewriteProtectionService:
    """Deliberately not wired into the public Rewrite selector."""

    def run_group(
        self,
        members: tuple[object, ...],
        baselines: dict[object, str],
        *,
        analyze: Callable[[], tuple[bool, bool]],
        rewrite: Callable[[dict[object, str]], dict[object, str]],
        evaluate: Callable[[dict[object, str]], _Phase8Metric],
        repair: Callable[[dict[object, str], int], dict[object, str]],
        max_repairs: int,
    ) -> tuple[tuple[object, str], ...] | None:
        """Return sealed keyed candidates only after a whole group succeeds."""
        outcome = _run_group_operation(
            members,
            baselines,
            analyze=analyze,
            rewrite=rewrite,
            evaluate=evaluate,
            repair=repair,
            max_repairs=max_repairs,
        )
        if outcome.state != "succeeded" or outcome.revisions is None:
            return None
        return tuple((member, outcome.revisions[member]) for member in members)
