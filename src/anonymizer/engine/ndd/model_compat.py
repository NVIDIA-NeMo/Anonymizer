# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helper for calling ModelFacade.generate in both sync and async engines.

In async engine mode, DataDesigner's HTTP client is async-only, so the sync
``model.generate()`` raises. Custom column generators that make direct model
calls need this wrapper to work under both engines.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop

logger = logging.getLogger(__name__)

_TIMEOUT = 300


def model_generate(model: Any, **kwargs: Any) -> tuple[Any, list]:
    """Call ``model.generate()`` or fall back to ``model.agenerate()``.

    In async engine mode the sync HTTP client is unavailable.  We detect
    this and schedule ``agenerate()`` on the persistent event loop that
    DD's async engine manages (the same loop the ``httpx.AsyncClient``
    is bound to).
    """
    try:
        return model.generate(**kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        if "async" not in msg and "sync" not in msg:
            raise

    loop = ensure_async_engine_loop()
    future = asyncio.run_coroutine_threadsafe(model.agenerate(**kwargs), loop)
    return future.result(timeout=_TIMEOUT)
