# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch-ownership handshake shared by the controller and managed server."""

from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from inference_service_compiler.models import Sha256Hex

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

LAUNCH_TOKEN_ENVIRONMENT_VARIABLE = "ANONYMIZER_INFERENCE_LAUNCH_TOKEN"
LAUNCH_OWNERSHIP_HEADER = "X-Anonymizer-Launch-Token"
LAUNCH_OWNERSHIP_PATH = "/_anonymizer/launch-ownership"
LAUNCH_OWNERSHIP_PROOF_FIELD = "launch_token_sha256"
LAUNCH_OWNERSHIP_MIDDLEWARE = "inference_service_compiler.launch_ownership.launch_ownership"


def launch_token_proof(token: str) -> Sha256Hex:
    """Return the non-secret proof expected from the launched server."""
    return Sha256Hex(hashlib.sha256(token.encode()).hexdigest())


async def launch_ownership(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Prove that this server inherited the controller's launch-scoped token.

    Installed as vLLM middleware; the native GLiNER2 server serves the same contract.
    """
    if request.url.path != LAUNCH_OWNERSHIP_PATH:
        return await call_next(request)

    from starlette.responses import JSONResponse

    expected = os.environ.get(LAUNCH_TOKEN_ENVIRONMENT_VARIABLE)
    supplied = request.headers.get(LAUNCH_OWNERSHIP_HEADER)
    if expected is None or supplied is None or not secrets.compare_digest(supplied, expected):
        return JSONResponse(status_code=404, content={"detail": "not found"})
    return JSONResponse(content={LAUNCH_OWNERSHIP_PROOF_FIELD: launch_token_proof(expected)})
