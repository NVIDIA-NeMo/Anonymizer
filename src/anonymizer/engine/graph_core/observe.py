# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observed-request channel: code running inside a graph step reports the requests it made.

The kernel opens one :class:`Activation` around each callable call. Code the callable reaches, on the same thread
or in a copied context, calls :func:`report_requests`; outside an activation the call does nothing.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator


class Activation:
    """Requests reported during one callable call; ``requests`` stays ``None`` until something reports."""

    def __init__(self) -> None:
        self.requests: int | None = None


_ACTIVATION: contextvars.ContextVar[Activation | None] = contextvars.ContextVar(
    "anonymizer_graph_activation", default=None
)


def report_requests(n: int) -> None:
    """Add ``n`` to the current activation's observed count; a no-op outside one."""
    activation = _ACTIVATION.get()
    if activation is not None:
        activation.requests = (activation.requests or 0) + n


@contextlib.contextmanager
def activation() -> Iterator[Activation]:
    """Open a fresh activation for the duration of one callable call."""
    current = Activation()
    token = _ACTIVATION.set(current)
    try:
        yield current
    finally:
        _ACTIVATION.reset(token)
